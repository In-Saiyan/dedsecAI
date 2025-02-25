import jwt, { JwtPayload, TokenExpiredError } from 'jsonwebtoken'
import crypto from 'crypto'
import ms from 'ms'

import connectDB from '@/lib/db'
import { TwoFactorToken } from '@/lib/models/auth.model'

export interface IPayload extends JwtPayload {
    email: string
}

export interface IError {
    error: string
}

export const isTokenError = (res: IPayload | IError): res is IError => {
    return (res as IError).error !== undefined
}

const secretKey: string = (process.env.JWT_SECRET as string) || ''

export const generateToken = async (
    payload: { email: string },
    expiresInTime: ms.StringValue = '1h'
) => {
    const options: jwt.SignOptions = { expiresIn: ms(expiresInTime) }
    return jwt.sign(payload, secretKey, options) // jwt.io
}

export const verifyToken = async (
    token: string
): Promise<IPayload | IError> => {
    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET!) as IPayload
        return decoded
    } catch (error) {
        if (error instanceof TokenExpiredError) {
            return { error: 'Token has expired!' }
        } else {
            return { error: 'Invalid token!' }
        }
    }
}

export const generateCode = async (email: string) => {
    const token = crypto.randomInt(100000, 1000000).toString() // generate a six-digit random number
    // console.log({token})
    const expires = new Date(new Date().getTime() + 5 * 60 * 1000) // 5 mins

    await connectDB()

    await TwoFactorToken.deleteOne({ email })

    const twoFactorToken = new TwoFactorToken({
        email,
        token,
        expires,
    })

    await twoFactorToken.save()

    return token
}
